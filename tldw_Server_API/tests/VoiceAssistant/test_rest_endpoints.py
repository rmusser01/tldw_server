# test_rest_endpoints.py
# Integration tests for Voice Assistant REST endpoints
#
#######################################################################################################################
import builtins
import importlib
import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User, get_request_user
from tldw_Server_API.app.core.VoiceAssistant import (
    save_voice_command,
    save_voice_session,
    record_voice_command_event,
)
from tldw_Server_API.app.core.VoiceAssistant.schemas import (
    ActionType,
    VoiceCommand,
    VoiceSessionContext,
    VoiceSessionState,
)


pytestmark = pytest.mark.integration


# Test fixtures

@pytest.fixture()
def mock_user():
    """Create a mock user for authentication."""
    return User(id=123, username="testuser", email="test@example.com", is_active=True)


@pytest.fixture()
def client_with_user(monkeypatch, mock_user, tmp_path):
    """Create a TestClient with authenticated user."""
    async def override_user():
        return mock_user

    # Route user DB base dir to temp path
    base_dir = tmp_path / "user_dbs"
    base_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("USER_DB_BASE_DIR", str(base_dir))

    # Set up a real per-user ChaChaNotes DB
    from tldw_Server_API.app.core.DB_Management.db_path_utils import DatabasePaths
    from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
    from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
    from tldw_Server_API.app.core.VoiceAssistant import (
        VoiceCommandRegistry,
        get_voice_command_registry,
        get_voice_command_router,
        get_voice_workflow_handler,
    )

    db_path = DatabasePaths.get_chacha_db_path(mock_user.id)
    db = CharactersRAGDB(db_path=str(db_path), client_id=str(mock_user.id))

    async def override_get_db():
        return db

    registry = VoiceCommandRegistry()
    router = MockVoiceCommandRouter()
    workflow_handler = MockVoiceWorkflowHandler()

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.voice_assistant.get_voice_command_registry",
        lambda: registry
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.voice_assistant.get_voice_command_router",
        lambda: router
    )
    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.voice_assistant.get_voice_workflow_handler",
        lambda: workflow_handler
    )

    # Mock TTS generation
    async def mock_generate_tts(text, provider=None, model=None, voice=None, response_format="mp3"):
        return b"audio-data", "audio/mpeg"

    monkeypatch.setattr(
        "tldw_Server_API.app.api.v1.endpoints.voice_assistant._generate_tts_audio",
        mock_generate_tts
    )

    # Voice routes are not mounted in the minimal test app profile.
    monkeypatch.setenv("MINIMAL_TEST_APP", "0")
    monkeypatch.setenv("ULTRA_MINIMAL_APP", "0")
    # Route toggles are cached; clear to ensure env changes are honored.
    try:
        from tldw_Server_API.app.core import config as core_config

        core_config._route_toggle_policy.cache_clear()
    except Exception:
        _ = None
    from tldw_Server_API.app import main as app_main

    importlib.reload(app_main)
    app = app_main.app
    app.dependency_overrides[get_request_user] = override_user
    app.dependency_overrides[get_chacha_db_for_user] = override_get_db

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client, registry, router, db

    app.dependency_overrides.clear()
    db.close_all_connections()


# Mock implementations
class MockVoiceCommandRouter:
    """Mock command router for testing."""

    def __init__(self):
        self.processed_commands: List[str] = []

    async def process_command(
        self,
        text: str,
        user_id: int,
        session_id: str,
        db: Any = None,
        persona_id: Optional[str] = None,
    ) -> tuple[Any, str]:
        from tldw_Server_API.app.core.VoiceAssistant.schemas import ActionResult, ActionType

        self.processed_commands.append(text)

        if session_id is None:
            session_id = f"session-{uuid.uuid4()}"

        result = ActionResult(
            success=True,
            action_type=ActionType.LLM_CHAT,
            response_text=f"Processed: {text}",
            result_data={"echo": text},
            execution_time_ms=10.0,
        )
        return result, session_id

    async def get_workflow_status(
        self, run_id: str, user_id: int
    ) -> Optional[Dict[str, Any]]:
        return None

    async def cancel_workflow(self, run_id: str, user_id: int) -> bool:
        return False


class MockVoiceWorkflowHandler:
    """Mock workflow handler for testing."""

    def get_voice_workflow_templates(self) -> Dict[str, Any]:
        return {
            "template-1": {
                "name": "Research Workflow",
                "metadata": {
                    "description": "Research a topic",
                    "voice_trigger": True,
                },
                "steps": [{"id": "step1"}, {"id": "step2"}],
            },
            "template-2": {
                "name": "Summary Workflow",
                "metadata": {
                    "description": "Summarize content",
                    "voice_trigger": True,
                },
                "steps": [{"id": "step1"}],
            },
        }


@pytest.mark.asyncio
async def test_voice_command_dry_run_sanitizes_module_import_errors(monkeypatch, mock_user):
    """Dry-run availability errors should not expose import/backend details."""
    from tldw_Server_API.app.api.v1.endpoints import voice_assistant as voice_mod

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "tldw_Server_API.app.core.VoiceAssistant.intent_parser":
            raise ImportError("voice backend exploded")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(HTTPException) as exc_info:
        await voice_mod.voice_command_dry_run(
            payload=voice_mod.VoiceCommandDryRunRequest(phrase="hello assistant"),
            request=object(),
            current_user=mock_user,
            db=object(),
        )

    assert exc_info.value.status_code == 501
    assert exc_info.value.detail == "Voice assistant module not available"


@pytest.mark.asyncio
async def test_voice_command_dry_run_sanitizes_unexpected_failures(monkeypatch, mock_user):
    """Dry-run backend failures should return a stable operational detail."""
    from tldw_Server_API.app.api.v1.endpoints import voice_assistant as voice_mod

    logged_errors = []

    class LoggerStub:
        def error(self, message, *args, **kwargs):
            logged_errors.append((message, args, kwargs))

    monkeypatch.setattr(voice_mod, "logger", LoggerStub())
    monkeypatch.setattr(
        voice_mod,
        "get_voice_command_registry",
        lambda: (_ for _ in ()).throw(RuntimeError("dry-run backend exploded")),
    )

    with pytest.raises(HTTPException) as exc_info:
        await voice_mod.voice_command_dry_run(
            payload=voice_mod.VoiceCommandDryRunRequest(phrase="hello assistant"),
            request=object(),
            current_user=mock_user,
            db=object(),
        )

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Dry-run failed"
    assert logged_errors == [("Voice command dry-run failed", (), {})]


# Test classes

class TestVoiceCommandEndpoint:
    """Tests for POST /api/v1/voice/command."""

    def test_process_command_success(self, client_with_user):
        """Test processing a text command."""
        client, _, router, _ = client_with_user

        response = client.post(
            "/api/v1/voice/command",
            json={
                "text": "search for machine learning",
                "include_tts": False,
            }
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "Processed:" in data["action_result"]["response_text"]
        assert data["session_id"] is not None

    def test_process_command_with_tts(self, client_with_user):
        """Test processing a command with TTS response."""
        client, _, _, _ = client_with_user

        response = client.post(
            "/api/v1/voice/command",
            json={
                "text": "hello assistant",
                "include_tts": True,
                "tts_format": "mp3",
            }
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["output_audio"] is not None  # Base64 encoded audio
        assert data["output_audio_format"] == "mp3"


class TestVoiceCommandsListEndpoint:
    """Tests for GET /api/v1/voice/commands."""

    def test_list_commands_success(self, client_with_user):
        """Test listing available voice commands."""
        client, registry, _, _ = client_with_user

        response = client.get("/api/v1/voice/commands")

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert "commands" in data
        assert "total" in data
        assert data["total"] >= 0

    def test_list_commands_exclude_system(self, client_with_user):
        """Test listing commands excluding system commands."""
        client, _, _, _ = client_with_user

        response = client.get(
            "/api/v1/voice/commands",
            params={"include_system": False}
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        # Should not include system commands (user_id=0)
        for cmd in data["commands"]:
            assert cmd["user_id"] != 0


class TestVoiceCommandsCreateEndpoint:
    """Tests for POST /api/v1/voice/commands."""

    def test_create_command_success(self, client_with_user, mock_user):
        """Test creating a new voice command."""
        client, registry, _, _ = client_with_user

        response = client.post(
            "/api/v1/voice/commands",
            json={
                "name": "my custom command",
                "phrases": ["do something", "perform action"],
                "action_type": "llm_chat",
                "action_config": {},
                "priority": 50,
                "enabled": True,
                "requires_confirmation": False,
                "description": "A custom command",
            }
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "my custom command"
        assert data["user_id"] == mock_user.id
        assert set(data["phrases"]) == {"do something", "perform action"}
        assert data["priority"] == 50

    def test_create_command_missing_phrases(self, client_with_user):
        """Test that creating a command without phrases fails."""
        client, _, _, _ = client_with_user

        response = client.post(
            "/api/v1/voice/commands",
            json={
                "name": "invalid command",
                "phrases": [],  # Empty phrases
                "action_type": "llm_chat",
            }
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 422  # Validation error


class TestVoiceCommandsDeleteEndpoint:
    """Tests for DELETE /api/v1/voice/commands/{command_id}."""

    def test_delete_command_success(self, client_with_user, mock_user):
        """Test deleting a user command."""
        client, _, _, db = client_with_user

        # First create a command
        cmd = VoiceCommand(
            id="user-cmd-1",
            user_id=mock_user.id,
            name="deletable",
            phrases=["delete me"],
            action_type=ActionType.LLM_CHAT,
            action_config={},
        )
        save_voice_command(db, cmd)

        response = client.delete("/api/v1/voice/commands/user-cmd-1")

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 204

    def test_delete_command_not_found(self, client_with_user):
        """Test deleting a non-existent command."""
        client, _, _, _ = client_with_user

        response = client.delete("/api/v1/voice/commands/nonexistent")

        if response.status_code == 405:  # Method not allowed if routes not available
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 404

    def test_delete_system_command_forbidden(self, client_with_user):
        """Test that deleting a system command is forbidden."""
        client, _, _, _ = client_with_user

        # System command has user_id=0
        response = client.delete("/api/v1/voice/commands/builtin-search-media")

        if response.status_code == 405:
            pytest.skip("Voice assistant routes not available")

        # Should either be forbidden or not found
        assert response.status_code in [403, 404]


class TestVoiceCommandDetailEndpoints:
    """Tests for GET/PUT/toggle voice command endpoints."""

    def test_get_command_success(self, client_with_user, mock_user):
        client, _, _, db = client_with_user

        cmd = VoiceCommand(
            id="detail-cmd-1",
            user_id=mock_user.id,
            name="detail command",
            phrases=["detail command"],
            action_type=ActionType.LLM_CHAT,
            action_config={},
            enabled=True,
        )
        save_voice_command(db, cmd)

        response = client.get("/api/v1/voice/commands/detail-cmd-1")

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "detail-cmd-1"
        assert data["name"] == "detail command"

    def test_update_command_preserves_enabled(self, client_with_user, mock_user):
        client, _, _, db = client_with_user

        cmd = VoiceCommand(
            id="update-cmd-1",
            user_id=mock_user.id,
            name="update command",
            phrases=["update me"],
            action_type=ActionType.LLM_CHAT,
            action_config={},
            enabled=False,
            priority=10,
        )
        save_voice_command(db, cmd)

        response = client.put(
            "/api/v1/voice/commands/update-cmd-1",
            json={
                "name": "updated command",
                "phrases": ["updated phrase"],
                "action_type": "llm_chat",
                "action_config": {},
                "priority": 5,
                "requires_confirmation": False,
                "description": "Updated",
            },
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "updated command"
        assert data["priority"] == 5
        # Enabled should remain false since it was not provided
        assert data["enabled"] is False

    def test_toggle_command_success(self, client_with_user, mock_user):
        client, _, _, db = client_with_user

        cmd = VoiceCommand(
            id="toggle-cmd-1",
            user_id=mock_user.id,
            name="toggle command",
            phrases=["toggle"],
            action_type=ActionType.LLM_CHAT,
            action_config={},
            enabled=True,
        )
        save_voice_command(db, cmd)

        response = client.post(
            "/api/v1/voice/commands/toggle-cmd-1/toggle",
            json={"enabled": False},
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["enabled"] is False


class TestVoiceSessionsEndpoint:
    """Tests for GET /api/v1/voice/sessions."""

    def test_list_sessions_empty(self, client_with_user):
        """Test listing sessions when none exist."""
        client, _, _, _ = client_with_user

        response = client.get("/api/v1/voice/sessions")

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["sessions"] == []
        assert data["total"] == 0

    def test_list_sessions_with_active(self, client_with_user, mock_user):
        """Test listing sessions when some exist."""
        client, _, _, db = client_with_user

        session = VoiceSessionContext(
            session_id=str(uuid.uuid4()),
            user_id=mock_user.id,
            state=VoiceSessionState.IDLE,
        )
        save_voice_session(db, session)

        response = client.get("/api/v1/voice/sessions")

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert len(data["sessions"]) == 1

    def test_list_sessions_includes_canonical_pagination(self, client_with_user, mock_user) -> None:
        """Voice session listing preserves total while exposing canonical offset pagination."""
        client, _, _, db = client_with_user
        now = datetime.utcnow()
        sessions = [
            VoiceSessionContext(
                session_id="voice-session-newest",
                user_id=mock_user.id,
                state=VoiceSessionState.IDLE,
                last_activity=now,
            ),
            VoiceSessionContext(
                session_id="voice-session-middle",
                user_id=mock_user.id,
                state=VoiceSessionState.IDLE,
                last_activity=now - timedelta(minutes=1),
            ),
            VoiceSessionContext(
                session_id="voice-session-oldest",
                user_id=mock_user.id,
                state=VoiceSessionState.IDLE,
                last_activity=now - timedelta(minutes=2),
            ),
        ]
        for session in sessions:
            save_voice_session(db, session)

        response = client.get(
            "/api/v1/voice/sessions",
            params={"active_only": False, "limit": 1, "offset": 1},
        )

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert [session["session_id"] for session in data["sessions"]] == ["voice-session-middle"]
        assert data["total"] == 3
        assert data["pagination"] == {
            "mode": "offset",
            "limit": 1,
            "offset": 1,
            "total": 3,
            "has_more": True,
            "next_offset": 2,
        }


class TestVoiceSessionDeleteEndpoint:
    """Tests for DELETE /api/v1/voice/sessions/{session_id}."""

    def test_end_session_success(self, client_with_user, mock_user):
        """Test ending an active session."""
        client, _, _, db = client_with_user

        session = VoiceSessionContext(
            session_id=str(uuid.uuid4()),
            user_id=mock_user.id,
            state=VoiceSessionState.IDLE,
        )
        save_voice_session(db, session)

        response = client.delete(f"/api/v1/voice/sessions/{session.session_id}")

        if response.status_code == 405:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 204

    def test_end_session_not_found(self, client_with_user):
        """Test ending a non-existent session."""
        client, _, _, _ = client_with_user

        response = client.delete("/api/v1/voice/sessions/nonexistent")

        if response.status_code == 405:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 404


class TestVoiceSessionDetailEndpoint:
    """Tests for GET /api/v1/voice/sessions/{session_id}."""

    def test_get_session_success(self, client_with_user, mock_user):
        client, _, _, db = client_with_user

        session = VoiceSessionContext(
            session_id=str(uuid.uuid4()),
            user_id=mock_user.id,
            state=VoiceSessionState.IDLE,
        )
        save_voice_session(db, session)

        response = client.get(f"/api/v1/voice/sessions/{session.session_id}")

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == session.session_id


class TestWorkflowTemplatesEndpoint:
    """Tests for GET /api/v1/voice/workflows/templates."""

    def test_list_workflow_templates(self, client_with_user):
        """Test listing available workflow templates."""
        client, _, _, _ = client_with_user

        response = client.get("/api/v1/voice/workflows/templates")

        if response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert "templates" in data
        assert data["total"] == 2
        assert len(data["templates"]) == 2

        # Check template structure
        template = data["templates"][0]
        assert "template_id" in template
        assert "name" in template
        assert "steps_count" in template


class TestVoiceAnalyticsEndpoints:
    """Tests for analytics and usage endpoints."""

    def test_voice_command_usage_and_analytics(self, client_with_user, mock_user):
        client, _, _, db = client_with_user

        cmd = VoiceCommand(
            id="analytics-cmd-1",
            user_id=mock_user.id,
            name="analytics command",
            phrases=["analytics"],
            action_type=ActionType.LLM_CHAT,
            action_config={},
            enabled=True,
        )
        save_voice_command(db, cmd)

        # Record two command events (1 success, 1 failure)
        record_voice_command_event(
            db,
            command_id=cmd.id,
            command_name=cmd.name,
            user_id=mock_user.id,
            action_type=ActionType.LLM_CHAT,
            success=True,
            response_time_ms=120.0,
            session_id="s1",
        )
        record_voice_command_event(
            db,
            command_id=cmd.id,
            command_name=cmd.name,
            user_id=mock_user.id,
            action_type=ActionType.LLM_CHAT,
            success=False,
            response_time_ms=200.0,
            session_id="s1",
        )

        usage_response = client.get(f"/api/v1/voice/commands/{cmd.id}/usage")
        if usage_response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert usage_response.status_code == 200
        usage = usage_response.json()
        assert usage["total_invocations"] == 2
        assert usage["success_count"] == 1
        assert usage["error_count"] == 1

        analytics_response = client.get("/api/v1/voice/analytics", params={"days": 7})
        if analytics_response.status_code == 404:
            pytest.skip("Voice assistant routes not available")

        assert analytics_response.status_code == 200
        analytics = analytics_response.json()
        assert analytics["total_commands_processed"] >= 2
        assert analytics["total_voice_commands"] >= 1
        assert analytics["enabled_commands"] >= 1


class TestWorkflowStatusEndpoint:
    """Tests for GET /api/v1/voice/workflows/{run_id}/status."""

    def test_get_workflow_status_not_found(self, client_with_user):
        """Test getting status of non-existent workflow."""
        client, _, _, _ = client_with_user

        response = client.get("/api/v1/voice/workflows/nonexistent/status")

        if response.status_code == 405:
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 404


class TestWorkflowCancelEndpoint:
    """Tests for POST /api/v1/voice/workflows/{run_id}/cancel."""

    def test_cancel_workflow_not_found(self, client_with_user):
        """Test canceling a non-existent workflow."""
        client, _, _, _ = client_with_user

        response = client.post("/api/v1/voice/workflows/nonexistent/cancel")

        if response.status_code in (404, 405):
            pytest.skip("Voice assistant routes not available")

        assert response.status_code == 200
        data = response.json()
        assert data["cancelled"] is False


#
# End of test_rest_endpoints.py
#######################################################################################################################
